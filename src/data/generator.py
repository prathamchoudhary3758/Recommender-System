"""
Synthetic Data Generation Engine for Two-Stage Recommender Systems.

Simulates realistic e-commerce/content interaction telemetry:
- Users with latent category preference clusters and historical engagement profiles.
- Items following a power-law (Pareto) popularity distribution across distinct categories.
- Temporal interactions with realistic click probabilities, dwell times, and timestamps.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd

from config.settings import DataConfig, get_default_config


class SyntheticDataGenerator:
    """Generates synthetic users, items, and temporal interaction telemetry."""

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or get_default_config().data
        self.rng = np.random.default_rng(self.config.random_seed)

    def generate_users(self) -> pd.DataFrame:
        """
        Generates user profiles with demographic tiers and category preference affinities.
        """
        num_users = self.config.num_users
        user_ids = [f"u_{i:06d}" for i in range(num_users)]
        
        # User activity clusters: 10% heavy, 30% medium, 60% casual
        activity_tiers = self.rng.choice(
            ["casual", "medium", "heavy"],
            size=num_users,
            p=[0.60, 0.30, 0.10],
        )
        
        # Age brackets
        age_groups = self.rng.choice(
            ["18-24", "25-34", "35-49", "50+"],
            size=num_users,
            p=[0.25, 0.40, 0.25, 0.10],
        )

        # Primary preferred category (0 to num_categories - 1)
        primary_categories = self.rng.integers(0, self.config.num_categories, size=num_users)

        # Base propensity to click (higher for heavy users)
        base_ctr = np.where(
            activity_tiers == "heavy",
            self.rng.uniform(0.15, 0.25, size=num_users),
            np.where(
                activity_tiers == "medium",
                self.rng.uniform(0.08, 0.15, size=num_users),
                self.rng.uniform(0.02, 0.08, size=num_users),
            ),
        )

        users_df = pd.DataFrame({
            "user_id": user_ids,
            "activity_tier": activity_tiers,
            "age_group": age_groups,
            "primary_category": primary_categories,
            "base_ctr": np.round(base_ctr, 4),
        })
        return users_df

    def generate_items(self) -> pd.DataFrame:
        """
        Generates item catalog with category assignments, quality scores, and popularity tiers.
        """
        num_items = self.config.num_items
        item_ids = [f"i_{i:05d}" for i in range(num_items)]
        
        categories = self.rng.integers(0, self.config.num_categories, size=num_items)
        
        # Popularity follows a power-law distribution (Pareto alpha=1.5)
        raw_popularity = self.rng.pareto(a=1.5, size=num_items)
        # Normalize popularity between 0 and 1
        popularity_scores = (raw_popularity - raw_popularity.min()) / (
            raw_popularity.max() - raw_popularity.min() + 1e-8
        )
        
        # Item intrinsic quality (normally distributed, 0 to 1)
        quality_scores = np.clip(self.rng.normal(loc=0.5, scale=0.18, size=num_items), 0.0, 1.0)
        
        # Day of introduction into the catalog (some are older, some newly introduced)
        intro_days = self.rng.integers(0, max(1, self.config.simulation_days // 4), size=num_items)

        items_df = pd.DataFrame({
            "item_id": item_ids,
            "category_id": categories,
            "popularity_score": np.round(popularity_scores, 4),
            "quality_score": np.round(quality_scores, 4),
            "intro_day": intro_days,
        })
        return items_df

    def generate_interactions(
        self, users_df: pd.DataFrame, items_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Simulates interaction events over the simulation time window.
        """
        num_users = len(users_df)
        num_items = len(items_df)
        total_days = self.config.simulation_days

        # Pre-calculate item sample probabilities based on popularity (power law)
        item_weights = items_df["popularity_score"].values + 0.05
        item_sampling_probs = item_weights / item_weights.sum()

        item_categories = items_df["category_id"].values
        item_qualities = items_df["quality_score"].values
        item_id_array = items_df["item_id"].values

        # Activity multipliers per tier
        tier_multipliers = {"casual": 15, "medium": 35, "heavy": 70}
        
        interaction_records = []
        user_ids = users_df["user_id"].values
        user_tiers = users_df["activity_tier"].values
        user_primary_cats = users_df["primary_category"].values
        user_base_ctrs = users_df["base_ctr"].values

        for idx in range(num_users):
            u_id = user_ids[idx]
            u_tier = user_tiers[idx]
            u_primary_cat = user_primary_cats[idx]
            u_base_ctr = user_base_ctrs[idx]

            # Sample interaction count for this user
            mean_interactions = tier_multipliers[u_tier]
            n_events = max(3, int(self.rng.normal(mean_interactions, mean_interactions * 0.25)))

            # Sample items for this user: 60% probability-weighted (popular items), 40% uniform
            n_weighted = int(n_events * 0.6)
            n_random = n_events - n_weighted
            
            chosen_item_indices = np.concatenate([
                self.rng.choice(num_items, size=n_weighted, p=item_sampling_probs, replace=True),
                self.rng.choice(num_items, size=n_random, replace=True),
            ])

            # Sample days across simulation window
            days = self.rng.integers(1, total_days + 1, size=n_events)
            # Sample seconds within day
            seconds_of_day = self.rng.integers(0, 86400, size=n_events)
            timestamps = days * 86400 + seconds_of_day

            for e_idx in range(n_events):
                item_idx = chosen_item_indices[e_idx]
                item_cat = item_categories[item_idx]
                item_qual = item_qualities[item_idx]
                item_id = item_id_array[item_idx]
                day = int(days[e_idx])
                ts = int(timestamps[e_idx])

                # Click logic: category affinity match gives +0.25 CTR boost
                category_match = 1 if (item_cat == u_primary_cat) else 0
                click_prob = u_base_ctr + (0.22 * category_match) + (0.15 * item_qual)
                click_prob = np.clip(click_prob, 0.01, 0.95)

                clicked = int(self.rng.uniform(0, 1) < click_prob)

                # Dwell time (longer if clicked)
                if clicked:
                    dwell_time = float(np.round(self.rng.exponential(scale=45.0) + 10.0, 1))
                    # Conversion / purchase: 15% of clicked items
                    conversion = int(self.rng.uniform(0, 1) < 0.15)
                    rating = int(self.rng.choice([3, 4, 5], p=[0.2, 0.4, 0.4]))
                else:
                    dwell_time = float(np.round(self.rng.exponential(scale=4.0) + 0.5, 1))
                    conversion = 0
                    rating = 0

                interaction_records.append((
                    u_id,
                    item_id,
                    ts,
                    day,
                    category_match,
                    clicked,
                    dwell_time,
                    conversion,
                    rating,
                ))

        interactions_df = pd.DataFrame(
            interaction_records,
            columns=[
                "user_id",
                "item_id",
                "timestamp",
                "day",
                "category_match",
                "clicked",
                "dwell_time_secs",
                "conversion",
                "rating",
            ],
        )

        # Sort chronologically by timestamp
        interactions_df.sort_values(by="timestamp", inplace=True, ignore_index=True)
        interactions_df.insert(0, "interaction_id", [f"ev_{i:08d}" for i in range(len(interactions_df))])
        return interactions_df

    def generate_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Convenience method to generate users, items, and interactions in one call."""
        users_df = self.generate_users()
        items_df = self.generate_items()
        interactions_df = self.generate_interactions(users_df, items_df)
        return users_df, items_df, interactions_df
