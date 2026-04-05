# Fraud Augmentation Pipeline

Generative AI for Real-Time Data Augmentation in Big Data Pipelines  
2-person college project | Ubuntu Linux | Python + Kafka + PySpark + CTGAN

---

## Project Overview

This project implements a streaming data pipeline that uses CTGAN (Conditional Tabular GAN)
to augment imbalanced fraud detection data in real time.
The augmented data is used to train better ML classifiers.

---

## Team Structure

| Person | Owns |
|--------|------|
| Person 1 | pipeline/, configs/, setup/, data ingestion, stream processing, storage |
| Person 2 | augmentation/, models/, evaluation/, dashboard/, CTGAN, metrics |

---

## Quickstart (Person 2 — after cloning)

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/fraud-augmentation-pipeline.git
cd fraud-augmentation-pipeline
```

### 2. Create Python environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Download dataset
Download `creditcard.csv` from:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
Place it at: `data/raw/creditcard.csv`

### 4. Start Kafka (Person 1 handles this — see setup/kafka_start.sh)

### 5. Run pipeline
```bash
# Terminal 1 — Start Kafka (Person 1)
bash setup/kafka_start.sh

# Terminal 2 — Start producer
python pipeline/producer.py

# Terminal 3 — Start stream processor
python pipeline/stream_processor.py

# Terminal 4 — Dashboard (Person 2)
streamlit run dashboard/app.py
```

---

## Folder Reference
data/raw/          — Place creditcard.csv here (gitignored)
data/processed/    — Clean Parquet batches (written by stream_processor.py)
data/augmented/    — Augmented Parquet batches (written after augmentation)
data/models/       — Saved CTGAN model .pkl file
data/sample/       — Sample data for testing without Kafka
pipeline/          — Kafka producer, PySpark stream processor, preprocessor
augmentation/      — CTGAN training, augment() function, SMOTE fallback
models/            — ML model training and prediction scripts
evaluation/        — Metrics, comparison tables, result charts
dashboard/         — Streamlit app
configs/           — config.yaml (all settings), schema.json (data contract)
setup/             — install.sh, kafka_start.sh
notebooks/         — EDA and offline experiments
docs/              — Architecture notes and results writeup

---

## The Integration Interface

Person 1's pipeline calls exactly one function from Person 2:
```python
from augmentation.augmentor import augment
augmented_df = augment(batch_df, batch_id)
```

**Input:** Pandas DataFrame with columns defined in `configs/schema.json`  
**Output:** Same DataFrame with synthetic rows appended + `is_synthetic` boolean column

---

## Key Config

All settings live in `configs/config.yaml`.  
Do not hardcode paths or parameters anywhere else.

---

## Branch Strategy
main              — stable only
feature/pipeline  — Person 1 works here
feature/augmentation — Person 2 works here
integration       — merge and test together
demo              — locked final version

---

## Dataset

Credit Card Fraud Detection  
Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
284,807 transactions | 492 fraud cases (0.17%) | 30 features  
Target column: `Class` (0 = normal, 1 = fraud)
