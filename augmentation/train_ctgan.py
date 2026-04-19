"""
Offline CTGAN Training Module for Fraud Data
This script trains a CTGAN model on the minority class (fraud cases)
so it can generate synthetic fraud records during the streaming pipeline.
"""

import pandas as pd
import yaml
import json
import os
import logging
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("train_ctgan")

def load_config():
    with open("configs/config.yaml", "r",encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_schema():
    with open("configs/schema.json", "r",encoding="utf-8") as f:
        return json.load(f)

def main():
    config = load_config()
    schema = load_schema()

    raw_path = config["paths"]["raw_data"]
    model_path = config["paths"]["ctgan_model"]

    if not os.path.exists(raw_path):
        log.error(f"Dataset not found at {raw_path}. Please place creditcard.csv in data/raw/")
        return

    log.info(f"Loading dataset from {raw_path}...")
    df = pd.read_csv(raw_path)

    # We only train CTGAN on the minority class to purely generate synthetic fraud rows
    target_col = schema["target_column"]
    minority_class = config["augmentation"]["minority_class"]
    
    fraud_df = df[df[target_col] == minority_class].copy()
    log.info(f"Extracted {len(fraud_df)} minority class records for training.")

    if len(fraud_df) == 0:
        log.error("No minority class records found in the dataset. Check target column and class value.")
        return

    # Define metadata for SDV
    log.info("Building metadata for SDV...")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(fraud_df)
    
    # Update id column if defined in schema to avoid treating it as a feature
    id_col = schema.get("id_column")
    if id_col in fraud_df.columns:
        metadata.update_column(column_name=id_col, sdtype='id')

    log.info("Initializing CTGAN Synthesizer...")
    # Using 100 epochs as a default for quick training, could be increased for better quality
    synthesizer = CTGANSynthesizer(metadata, epochs=100)

    log.info("Training CTGAN model... (This may take a few minutes)")
    synthesizer.fit(fraud_df)
    
    # Note: sdv library has a save method for synthesizers
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    synthesizer.save(model_path)
    log.info(f"CTGAN model saved successfully to {model_path}.")

if __name__ == "__main__":
    main()
