"""
src/evaluation/splits.py

Leakage-aware data partitioning strategies:
1. Spatial Block Holdout (primary): Partitions the study area into spatial tiles,
   holding out entire geographic blocks to prevent near-duplicate leakage.
2. Event Holdout: Trains on one flood event (e.g., June 2015) and evaluates on a separate
   unseen flood event (e.g., August 2015) in the same geographic region.
"""

from typing import Tuple, List, Dict
import pandas as pd
import numpy as np


def get_spatial_block_split(df: pd.DataFrame,
                            train_blocks: List[int],
                            val_blocks: List[int],
                            test_blocks: List[int],
                            event_filter: str = "DFO_4259") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a strict spatial block holdout split on valid land cells.
    Guarantees zero geographic overlap between train, validation, and test sets.
    """
    # Filter to the designated event and binary land cells (excluding permanent water and nodata)
    valid_df = df[(df["event_id"] == event_filter) & (df["flood_label"].isin([0.0, 1.0]))].copy()

    train_set = valid_df[valid_df["spatial_block_id"].isin(train_blocks)].copy()
    val_set = valid_df[valid_df["spatial_block_id"].isin(val_blocks)].copy()
    test_set = valid_df[valid_df["spatial_block_id"].isin(test_blocks)].copy()

    # Integrity verification: Assert disjoint block sets
    train_b = set(train_set["spatial_block_id"].unique())
    val_b = set(val_set["spatial_block_id"].unique())
    test_b = set(test_set["spatial_block_id"].unique())

    assert len(train_b.intersection(val_b)) == 0, f"Spatial overlap between train and val: {train_b.intersection(val_b)}"
    assert len(train_b.intersection(test_b)) == 0, f"Spatial overlap between train and test: {train_b.intersection(test_b)}"
    assert len(val_b.intersection(test_b)) == 0, f"Spatial overlap between val and test: {val_b.intersection(test_b)}"

    return train_set, val_set, test_set


def get_event_holdout_split(df: pd.DataFrame,
                            train_event: str = "DFO_4259",
                            test_event: str = "DFO_4288",
                            val_ratio: float = 0.15,
                            random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform an event holdout split: train on one event, evaluate on a separate future event.
    """
    train_full = df[(df["event_id"] == train_event) & (df["flood_label"].isin([0.0, 1.0]))].copy()
    test_set = df[(df["event_id"] == test_event) & (df["flood_label"].isin([0.0, 1.0]))].copy()

    # Create validation split from distinct spatial blocks of the training event
    unique_blocks = train_full["spatial_block_id"].unique()
    rng = np.random.RandomState(random_state)
    val_block_count = max(1, int(len(unique_blocks) * val_ratio))
    val_blocks = rng.choice(unique_blocks, size=val_block_count, replace=False)
    train_blocks = [b for b in unique_blocks if b not in val_blocks]

    train_set = train_full[train_full["spatial_block_id"].isin(train_blocks)].copy()
    val_set = train_full[train_full["spatial_block_id"].isin(val_blocks)].copy()

    return train_set, val_set, test_set


def get_random_stratified_split(df: pd.DataFrame,
                                event_filter: str = "DFO_4259",
                                train_ratio: float = 0.70,
                                val_ratio: float = 0.15,
                                random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a naive stratified random split across valid cells.
    Serves as an empirical baseline to measure performance inflation caused
    by spatial autocorrelation leakage relative to spatial block holdouts.
    """
    from sklearn.model_selection import train_test_split

    valid_df = df[(df["event_id"] == event_filter) & (df["flood_label"].isin([0.0, 1.0]))].copy()

    test_ratio = 1.0 - (train_ratio + val_ratio)
    train_val_df, test_df = train_test_split(
        valid_df,
        test_size=test_ratio,
        stratify=valid_df["flood_label"],
        random_state=random_state
    )

    val_share = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_share,
        stratify=train_val_df["flood_label"],
        random_state=random_state
    )

    return train_df, val_df, test_df
