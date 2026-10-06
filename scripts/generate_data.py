"""
CLI Utility to generate and save synthetic interaction datasets.

Usage:
    python scripts/generate_data.py [--users 5000] [--items 1000] [--days 28]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DataConfig, get_default_config
from src.data.generator import SyntheticDataGenerator
from src.data.splitter import TemporalSplitter
from src.data.sampler import NegativeSampler


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic recommender datasets.")
    parser.add_argument("--users", type=int, default=5_000, help="Number of users to generate.")
    parser.add_argument("--items", type=int, default=1_000, help="Number of items in catalog.")
    parser.add_argument("--days", type=int, default=28, help="Simulation time horizon in days.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    config = get_default_config()
    data_config = DataConfig(
        num_users=args.users,
        num_items=args.items,
        simulation_days=args.days,
        random_seed=args.seed,
    )

    print(f"[*] Generating synthetic dataset: {args.users} users, {args.items} items, {args.days} days...")
    generator = SyntheticDataGenerator(data_config)
    users_df, items_df, interactions_df = generator.generate_all()

    print(f"[+] Generated {len(users_df):,} users and {len(items_df):,} items.")
    print(f"[+] Simulated {len(interactions_df):,} interaction events.")

    # Temporal split
    print("[*] Performing temporal train/val/test split...")
    splitter = TemporalSplitter(data_config)
    train_df, val_df, test_df = splitter.split(interactions_df)
    summary = splitter.get_split_summary(train_df, val_df, test_df)

    print(f"    - Train events: {summary['train_events']:,} ({summary['train_ratio']*100:.1f}%) | Days: {summary['train_day_range']}")
    print(f"    - Val events:   {summary['val_events']:,} ({summary['val_ratio']*100:.1f}%) | Days: {summary['val_day_range']}")
    print(f"    - Test events:  {summary['test_events']:,} ({summary['test_ratio']*100:.1f}%) | Days: {summary['test_day_range']}")

    # Negative sampling demonstration on train set
    print("[*] Augmenting training set with negative samples (ratio 2:1)...")
    sampler = NegativeSampler(
        all_item_ids=items_df["item_id"].tolist(),
        item_popularity_weights=items_df["popularity_score"].values,
        random_seed=args.seed,
    )
    augmented_train_df = sampler.augment_interactions_with_negatives(
        train_df, num_negatives_per_positive=2, strategy="popularity"
    )
    print(f"[+] Augmented train set size: {len(augmented_train_df):,} events (Positives + Negatives).")

    # Output directory
    output_dir = config.processed_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    users_df.to_parquet(output_dir / "users.parquet", index=False)
    items_df.to_parquet(output_dir / "items.parquet", index=False)
    augmented_train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)

    print(f"[SUCCESS] All datasets saved to: {output_dir}")


if __name__ == "__main__":
    main()
