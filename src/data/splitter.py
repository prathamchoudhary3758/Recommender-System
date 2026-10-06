"""
Temporal Data Splitting Engine for Recommender Systems.

Enforces strict time-based boundaries to prevent lookahead data leakage:
- Train Split: Historical interactions (Days 1 to train_end_day).
- Validation Split: Intermediate window (Days train_end_day+1 to val_end_day).
- Test Split: Future evaluation window (Days val_end_day+1 to test_end_day).
"""

from typing import Tuple, Dict, Any, Optional
import pandas as pd

from config.settings import DataConfig, get_default_config


class TemporalSplitter:
    """Splits interaction events strictly across temporal time horizons."""

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or get_default_config().data

    def split(
        self, interactions_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits interactions into (train_df, val_df, test_df) based on day boundaries.

        Returns:
            train_df: Interactions where day <= train_end_day
            val_df: Interactions where train_end_day < day <= val_end_day
            test_df: Interactions where day > val_end_day
        """
        if "day" not in interactions_df.columns:
            raise ValueError("Input interactions_df must contain a 'day' column.")

        train_cutoff = self.config.train_end_day
        val_cutoff = self.config.val_end_day

        train_df = interactions_df[interactions_df["day"] <= train_cutoff].copy()
        val_df = interactions_df[
            (interactions_df["day"] > train_cutoff) & (interactions_df["day"] <= val_cutoff)
        ].copy()
        test_df = interactions_df[interactions_df["day"] > val_cutoff].copy()

        # Reset indices
        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        return train_df, val_df, test_df

    def get_split_summary(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Computes diagnostic statistics on the temporal splits.
        """
        total_events = len(train_df) + len(val_df) + len(test_df)
        all_train_users = set(train_df["user_id"])
        all_train_items = set(train_df["item_id"])

        val_users = set(val_df["user_id"])
        val_items = set(val_df["item_id"])
        test_users = set(test_df["user_id"])
        test_items = set(test_df["item_id"])

        summary = {
            "total_events": total_events,
            "train_events": len(train_df),
            "val_events": len(val_df),
            "test_events": len(test_df),
            "train_ratio": round(len(train_df) / total_events, 4) if total_events > 0 else 0,
            "val_ratio": round(len(val_df) / total_events, 4) if total_events > 0 else 0,
            "test_ratio": round(len(test_df) / total_events, 4) if total_events > 0 else 0,
            "train_day_range": (train_df["day"].min(), train_df["day"].max()) if len(train_df) else None,
            "val_day_range": (val_df["day"].min(), val_df["day"].max()) if len(val_df) else None,
            "test_day_range": (test_df["day"].min(), test_df["day"].max()) if len(test_df) else None,
            # User & Item overlap / cold-start checks
            "val_user_coverage": round(len(val_users & all_train_users) / max(1, len(val_users)), 4),
            "test_user_coverage": round(len(test_users & all_train_users) / max(1, len(test_users)), 4),
            "val_item_coverage": round(len(val_items & all_train_items) / max(1, len(val_items)), 4),
            "test_item_coverage": round(len(test_items & all_train_items) / max(1, len(test_items)), 4),
        }
        return summary
