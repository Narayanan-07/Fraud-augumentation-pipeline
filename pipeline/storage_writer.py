"""
Storage Writer — Saves processed and augmented batches as Parquet.
Person 1 owns this file.
"""

import logging
import pandas as pd
from pathlib import Path

log = logging.getLogger(__name__)


def write_batch(df: pd.DataFrame, batch_id: int, output_dir: str,
                label: str = "processed") -> str:
    """
    Write a DataFrame batch to Parquet.
    Returns the path written to.
    """
    if df.empty:
        log.warning(f"Batch {batch_id} is empty — skipping write.")
        return ""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"batch_{batch_id:05d}_{label}.parquet"
    filepath = out_dir / filename

    df.to_parquet(filepath, index=False, engine="pyarrow")
    log.info(f"[{label.upper()}] Batch {batch_id}: {len(df)} rows → {filepath}")
    return str(filepath)


def write_processed(df: pd.DataFrame, batch_id: int, config: dict) -> str:
    return write_batch(df, batch_id, config["paths"]["processed_dir"], "processed")


def write_augmented(df: pd.DataFrame, batch_id: int, config: dict) -> str:
    return write_batch(df, batch_id, config["paths"]["augmented_dir"], "augmented")
