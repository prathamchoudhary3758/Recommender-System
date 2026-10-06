"""
Negative Sampling Engine for Two-Stage Recommender Systems.

Generates negative examples for candidate retrieval and ranking models:
- Random Negative Sampling: Samples unobserved items uniformly from the catalog.
- Popularity-Biased (Hard) Negative Sampling: Samples unobserved items proportional 
  to item catalog popularity.
"""

from typing import Set, Dict, List, Optional
import numpy as np
import pandas as pd


class NegativeSampler:
    """Samples negative (unobserved) items for user interaction events."""

    def __init__(
        self,
        all_item_ids: List[str],
        item_popularity_weights: Optional[np.ndarray] = None,
        random_seed: int = 42,
    ):
        self.all_item_ids = np.array(all_item_ids)
        self.num_items = len(self.all_item_ids)
        self.rng = np.random.default_rng(random_seed)

        if item_popularity_weights is not None:
            weights = np.array(item_popularity_weights, dtype=np.float64)
            weights = np.clip(weights, 1e-6, None)
            self.item_probs = weights / weights.sum()
        else:
            self.item_probs = np.full(self.num_items, 1.0 / self.num_items)

        # Mapping for O(1) index lookup
        self.item_to_idx = {item_id: idx for idx, item_id in enumerate(self.all_item_ids)}

    def sample_negatives_for_user(
        self,
        observed_items: Set[str],
        num_negatives: int = 4,
        strategy: str = "random",
    ) -> List[str]:
        """
        Samples unobserved negative items for a single user.

        Args:
            observed_items: Set of items the user has already interacted with.
            num_negatives: Number of negative items to sample.
            strategy: 'random' (uniform) or 'popularity' (popularity-weighted).

        Returns:
            List of sampled negative item IDs.
        """
        if len(observed_items) >= self.num_items:
            return []

        sampled_negatives: List[str] = []
        max_attempts = num_negatives * 15
        attempts = 0

        p = self.item_probs if strategy == "popularity" else None

        while len(sampled_negatives) < num_negatives and attempts < max_attempts:
            # Over-sample candidates in batches for efficiency
            needed = (num_negatives - len(sampled_negatives)) * 2
            candidate_indices = self.rng.choice(self.num_items, size=needed, replace=True, p=p)
            candidate_items = self.all_item_ids[candidate_indices]

            for item in candidate_items:
                if item not in observed_items and item not in sampled_negatives:
                    sampled_negatives.append(item)
                    if len(sampled_negatives) == num_negatives:
                        break
            attempts += needed

        # Fallback: uniform fill if max attempts reached
        if len(sampled_negatives) < num_negatives:
            unobserved = [i for i in self.all_item_ids if i not in observed_items and i not in sampled_negatives]
            if unobserved:
                fill_count = min(num_negatives - len(sampled_negatives), len(unobserved))
                sampled_negatives.extend(self.rng.choice(unobserved, size=fill_count, replace=False).tolist())

        return sampled_negatives

    def augment_interactions_with_negatives(
        self,
        interactions_df: pd.DataFrame,
        num_negatives_per_positive: int = 4,
        strategy: str = "random",
    ) -> pd.DataFrame:
        """
        Augments an interaction DataFrame with negative rows (clicked=0).

        Args:
            interactions_df: DataFrame containing positive interactions (user_id, item_id, etc.).
            num_negatives_per_positive: Ratio of negatives to sample per positive interaction.
            strategy: 'random' or 'popularity'.

        Returns:
            A combined DataFrame with both positive (label=1) and negative (label=0) samples.
        """
        # Map observed items per user for quick filtering
        user_history: Dict[str, Set[str]] = (
            interactions_df.groupby("user_id")["item_id"].apply(set).to_dict()
        )

        negative_rows = []
        # Filter for positive interactions (clicked == 1)
        positive_df = interactions_df[interactions_df["clicked"] == 1]

        for _, row in positive_df.iterrows():
            user_id = row["user_id"]
            user_obs = user_history.get(user_id, set())
            neg_items = self.sample_negatives_for_user(
                observed_items=user_obs,
                num_negatives=num_negatives_per_positive,
                strategy=strategy,
            )

            for neg_item in neg_items:
                neg_row = {
                    "interaction_id": f"neg_{len(negative_rows):08d}",
                    "user_id": user_id,
                    "item_id": neg_item,
                    "timestamp": row["timestamp"],
                    "day": row["day"],
                    "category_match": 0,
                    "clicked": 0,
                    "dwell_time_secs": 0.0,
                    "conversion": 0,
                    "rating": 0,
                }
                negative_rows.append(neg_row)

        neg_df = pd.DataFrame(negative_rows)
        augmented_df = pd.concat([interactions_df, neg_df], ignore_index=True)
        augmented_df.sort_values(by="timestamp", inplace=True, ignore_index=True)
        return augmented_df
