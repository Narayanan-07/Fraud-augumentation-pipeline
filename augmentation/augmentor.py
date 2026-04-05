"""
Augmentation Module — Person 2 owns and implements this file.
─────────────────────────────────────────────────────────────
Interface contract:
  Input:  pandas DataFrame with columns from configs/schema.json (processed_columns)
  Output: pandas DataFrame — same columns + is_synthetic (bool) + batch_id (int)
          synthetic rows appended to real rows

Person 1's stream_processor.py calls:
  from augmentation.augmentor import augment
  augmented_df = augment(batch_df, batch_id)
"""

import pandas as pd
import logging

log = logging.getLogger(__name__)


def augment(df: pd.DataFrame, batch_id: int) -> pd.DataFrame:
    """
    PLACEHOLDER — Person 2: replace this implementation.

    Current behavior: returns input unchanged with required metadata columns.
    """
    log.warning(f"Batch {batch_id}: using PLACEHOLDER augmentor — no synthetic data added.")

    df = df.copy()
    df["is_synthetic"] = False
    df["batch_id"] = batch_id
    return df
