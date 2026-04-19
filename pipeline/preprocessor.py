"""
Preprocessor — Cleans and scales each micro-batch.
Person 1 owns this file.
Called by stream_processor.py on each batch DataFrame.
"""

import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


def load_schema(schema_path: str = "configs/schema.json") -> dict:
    with open(schema_path, "r") as f:
        return json.load(f)


def clean_batch(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Remove nulls, fix dtypes, drop irrelevant columns."""
    # Drop Time column (not a feature)
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])

    # Keep only schema columns that exist in df
    expected = schema["feature_columns"] + [schema["target_column"]]
    available = [c for c in expected if c in df.columns]
    df = df[available].copy()

    # Cast to numeric, drop rows that fail
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    # Cast Class to int
    df[schema["target_column"]] = df[schema["target_column"]].astype(int)

    return df


def scale_amount(df: pd.DataFrame, scaler_path: str) -> pd.DataFrame:
    """Scale the Amount column. Fit scaler on first batch, reuse after."""
    scaler_file = Path(scaler_path)

    if scaler_file.exists():
        with open(scaler_file, "rb") as f:
            scaler = pickle.load(f)
        df["Amount_scaled"] = scaler.transform(df[["Amount"]])
    else:
        scaler = StandardScaler()
        df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
        scaler_file.parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_file, "wb") as f:
            pickle.dump(scaler, f)
        log.info(f"Scaler fitted and saved to {scaler_path}")

    df = df.drop(columns=["Amount"])
    return df


def preprocess(df: pd.DataFrame, config: dict, schema: dict) -> pd.DataFrame:
    """Full preprocessing pipeline for one micro-batch."""
    if df.empty:
        log.warning("Empty batch received — skipping preprocessing.")
        return df

    df = clean_batch(df, schema)
    df = scale_amount(df, config["paths"]["scaler_model"])

    log.info(f"Preprocessed batch: {len(df)} rows | "
             f"Fraud: {df['Class'].sum()} | Normal: {(df['Class']==0).sum()}")
    return df
