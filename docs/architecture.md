# System Architecture

## Pipeline Flowcreditcard.csv
→ Kafka Producer (pipeline/producer.py)
→ Kafka Topic: fraud-stream
→ PySpark Structured Streaming (pipeline/stream_processor.py)
→ Preprocessor (pipeline/preprocessor.py)
→ [data/processed/.parquet]
→ Augmentor (augmentation/augmentor.py)  ← Person 2
→ [data/augmented/.parquet]
→ Model Training (models/train_model.py) ← Person 2
→ Evaluation (evaluation/evaluate.py)    ← Person 2
→ Dashboard (dashboard/app.py)           ← Person 2

## Component Owners
| Component | File | Owner |
|-----------|------|-------|
| Kafka Producer | pipeline/producer.py | Person 1 |
| Stream Processor | pipeline/stream_processor.py | Person 1 |
| Preprocessor | pipeline/preprocessor.py | Person 1 |
| Storage Writer | pipeline/storage_writer.py | Person 1 |
| Augmentor | augmentation/augmentor.py | Person 2 |
| Model Training | models/train_model.py | Person 2 |
| Evaluation | evaluation/evaluate.py | Person 2 |
| Dashboard | dashboard/app.py | Person 2 |
