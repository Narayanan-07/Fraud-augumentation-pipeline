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

Bug Fixes Applied
─────────────────
Bug 1: CTGAN model now uses lazy loading inside augment() via _get_synthesizer().
       The module-level eager load block is removed. Every call retries if the
       model wasn't available at startup, and logs a success message on retry.

Bug 2: min_samples_to_augment changed to 1 in configs/config.yaml so that even
       micro-batches with a single fraud record trigger augmentation.

Bug 3: SMOTE is_synthetic assignment now uses iloc after resetting the index so
       that positional indexing is safe regardless of the original index state.
"""

import pandas as pd
import logging
import yaml
import json
from pathlib import Path

log = logging.getLogger(__name__)

# ── Load config & schema once at module level ────────────────────────────────
with open("configs/config.yaml", "r") as _f:
    config = yaml.safe_load(_f)

with open("configs/schema.json", "r") as _f:
    schema = json.load(_f)

# ── Bug 1 Fix: Lazy CTGAN loader — retried on every augment() call ───────────
_ctgan_synthesizer = None          # starts as None; loaded on first real need
_ctgan_load_attempted = False      # avoid spamming log when model truly absent


def _get_synthesizer():
    """
    Return the CTGAN synthesizer, loading (or retrying to load) it from disk
    each time this function is called and the model is not yet in memory.
    Logs a clear success message the first time the model loads successfully
    after a failed startup attempt.
    """
    global _ctgan_synthesizer, _ctgan_load_attempted

    if _ctgan_synthesizer is not None:
        return _ctgan_synthesizer            # already loaded — fast path

    model_path = Path(config["paths"]["ctgan_model"])

    if not model_path.exists():
        if not _ctgan_load_attempted:
            log.warning(
                f"[CTGAN] Model not found at {model_path}. "
                "Will retry on next batch. Run train_ctgan.py to generate it."
            )
        _ctgan_load_attempted = True
        return None

    try:
        from sdv.single_table import CTGANSynthesizer
        _ctgan_synthesizer = CTGANSynthesizer.load(str(model_path))
        # Bug 1 fix: distinct success log so operators know the retry worked
        log.info(
            f"[CTGAN] ✅ Model loaded successfully from {model_path} "
            f"(lazy retry={'yes' if _ctgan_load_attempted else 'no'})."
        )
        _ctgan_load_attempted = True
        return _ctgan_synthesizer
    except Exception as exc:
        log.error(f"[CTGAN] Failed to load model: {exc}")
        _ctgan_load_attempted = True
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def augment(df: pd.DataFrame, batch_id: int) -> pd.DataFrame:
    """
    Applies data augmentation to the micro-batch based on config.

    Parameters
    ----------
    df       : processed micro-batch DataFrame (schema = processed_columns)
    batch_id : monotonically increasing integer passed by stream_processor

    Returns
    -------
    DataFrame with same columns + is_synthetic (bool) + batch_id (int).
    Real rows have is_synthetic=False; synthetic rows have is_synthetic=True.
    """
    df = df.copy()

    # ── Tracking columns ──────────────────────────────────────────────────────
    if "is_synthetic" not in df.columns:
        df["is_synthetic"] = False
    if "batch_id" not in df.columns:
        df["batch_id"] = batch_id

    method         = config["augmentation"]["method"]
    min_samples    = config["augmentation"]["min_samples_to_augment"]   # now 1
    target_col     = schema["target_column"]
    minority_class = config["augmentation"]["minority_class"]
    synthetic_ratio = config["augmentation"]["synthetic_ratio"]

    minority_count = int((df[target_col] == minority_class).sum())

    # ── Guard: skip if method is none or not enough minority rows ─────────────
    if method == "none" or minority_count < min_samples:
        log.info(
            f"Batch {batch_id}: Skipping augmentation "
            f"(method={method}, minority_count={minority_count}, "
            f"min_required={min_samples})."
        )
        return df

    num_to_generate = int(minority_count * synthetic_ratio)
    if num_to_generate <= 0:
        return df

    log.info(
        f"Batch {batch_id}: Augmenting with {method.upper()} — "
        f"generating {num_to_generate} rows (minority_count={minority_count})."
    )

    synthetic_df = None

    # ── CTGAN branch ──────────────────────────────────────────────────────────
    if method == "ctgan":
        synthesizer = _get_synthesizer()          # Bug 1: lazy load on every call
        if synthesizer is not None:
            try:
                synthetic_df = synthesizer.sample(num_rows=num_to_generate)
            except Exception as exc:
                log.error(f"Batch {batch_id}: CTGAN generation failed: {exc}")
        else:
            log.warning(
                f"Batch {batch_id}: CTGAN synthesizer unavailable — "
                "skipping augmentation for this batch."
            )

    # ── SMOTE branch ──────────────────────────────────────────────────────────
    elif method == "smote":
        try:
            from imblearn.over_sampling import SMOTE

            # k_neighbors must be < minority_count; minimum 1
            k_neighbors = max(1, min(minority_count - 1, 5))

            smote = SMOTE(
                sampling_strategy="minority",
                k_neighbors=k_neighbors,
                random_state=config["model"].get("random_state", 42),
            )

            feature_cols = [
                c for c in df.columns
                if c not in [target_col, "is_synthetic", "batch_id"]
            ]

            X_res, y_res = smote.fit_resample(df[feature_cols], df[target_col])

            original_len = len(df)

            # Bug 3 Fix: build res_df with a clean 0-based RangeIndex, then
            # use iloc so positional assignment is safe after fit_resample
            # resets the index internally.
            res_df = pd.DataFrame(X_res, columns=feature_cols)
            res_df[target_col]      = y_res.values
            res_df["batch_id"]      = batch_id
            res_df["is_synthetic"]  = False                          # default all real
            res_df.iloc[             # mark only the SMOTE-added rows as synthetic
                original_len:,
                res_df.columns.get_loc("is_synthetic"),
            ] = True

            log.info(
                f"Batch {batch_id}: SMOTE complete — "
                f"{len(res_df) - original_len} synthetic rows added."
            )
            return res_df

        except Exception as exc:
            log.error(f"Batch {batch_id}: SMOTE generation failed: {exc}")
            return df

    # ── Concatenate CTGAN synthetic rows ──────────────────────────────────────
    if synthetic_df is not None:
        synthetic_df = synthetic_df.copy()
        synthetic_df["is_synthetic"] = True
        synthetic_df["batch_id"]     = batch_id

        # Align columns: ensure synthetic_df has exactly the same columns as df
        for col in df.columns:
            if col not in synthetic_df.columns:
                synthetic_df[col] = df[col].iloc[0] if not df.empty else None

        df = pd.concat([df, synthetic_df[df.columns]], ignore_index=True)
        log.info(
            f"Batch {batch_id}: {len(synthetic_df)} synthetic rows appended via CTGAN."
        )

    return df
