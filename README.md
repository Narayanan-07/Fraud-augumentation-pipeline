# Real-Time Data Augmentation in Big Data Pipelines Using Generative AI

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Kafka](https://img.shields.io/badge/Kafka-3.7.0-orange) ![Spark](https://img.shields.io/badge/Spark-3.5.1-yellow) ![Docker](https://img.shields.io/badge/Docker-Compose-green) ![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

## 📜 Project Description

This project, **Fraud Augmentation Pipeline**, is a containerized real-time streaming pipeline designed for fraud detection. It leverages **CTGAN** (Conditional Tabular GAN) to augment imbalanced datasets in real time, enhancing the performance of machine learning classifiers. The pipeline processes streaming data, generates synthetic records, and provides live monitoring through an interactive dashboard.

---

## 🚀 Features

1. **Real-Time Streaming**: Kafka producer streams credit card fraud data at 50 records/second.
2. **Data Preprocessing**: PySpark cleans and scales data in 10-second micro-batches.
3. **Synthetic Data Generation**: CTGAN generates synthetic fraud records to balance the dataset.
4. **Data Storage**: Augmented data is saved as Parquet files for further analysis.
5. **Model Evaluation**: Offline evaluation of Random Forest, XGBoost, and Logistic Regression.
6. **Interactive Dashboard**: Streamlit dashboard visualizes pipeline metrics and evaluation results.

---

## 🛠️ Tech Stack

- **Programming Language**: Python 3.10
- **Streaming**: Apache Kafka 3.7.0
- **Processing**: Apache Spark 3.5.1
- **Data Augmentation**: CTGAN (via SDV library)
- **Machine Learning**: Scikit-learn, XGBoost
- **Visualization**: Streamlit, Plotly
- **Containerization**: Docker Compose

---

## 📂 Folder Structure

```plaintext
fraud-augmentation-pipeline/
├── data/
│   ├── raw/                  # Place creditcard.csv here (gitignored)
│   ├── processed/            # PySpark writes clean Parquet batches here
│   ├── augmented/            # PySpark writes augmented Parquet batches here
│   └── models/               # Place ctgan_model.pkl here after training
├── pipeline/
│   ├── producer.py           # Kafka CSV stream replay
│   ├── stream_processor.py   # PySpark Structured Streaming + foreachBatch
│   ├── preprocessor.py       # Cleaning, scaling, encoding
│   └── storage_writer.py     # Parquet writer
├── augmentation/
│   ├── train_ctgan.py        # Offline CTGAN training script
│   └── augmentor.py          # augment(df, batch_id) -> df interface
├── models/
│   └── train_eval.py         # Trains 3 classifiers, saves metrics.json
├── evaluation/
│   ├── metrics.py            # evaluate_model() and save_metrics() functions
│   └── results/              # metrics.json written here
├── dashboard/
│   └── app.py                # Streamlit dashboard, auto-refresh 5s
├── configs/
│   ├── config.yaml           # All settings (broker, paths, augmentation params)
│   └── schema.json           # Column names and dtypes contract
├── setup/
│   ├── install.sh            # Ubuntu local setup script
│   ├── kafka_start.sh        # Local Kafka startup
│   └── kafka_stop.sh         # Local Kafka stop
├── docker/
│   └── Dockerfile.pipeline   # Single Dockerfile for all Python services
├── docker-compose.yaml       # All 5 services
├── requirements.txt          # All Python dependencies
└── .gitignore                # Ignores data/raw, data/processed, data/augmented, models
```

---

## 🐳 Docker Services

- **zookeeper**: Manages Kafka cluster metadata.
- **kafka**: Apache Kafka broker for streaming data.
- **producer**: Custom Python service for streaming credit card data.
- **stream-processor**: PySpark service for data preprocessing and augmentation.
- **dashboard**: Streamlit dashboard for real-time monitoring.

All services run on a user-defined bridge network `pipeline-net`.

---

## 🏗️ Setup and Usage

### Step 1: Clone the Repository
```bash
git clone https://github.com/Narayanan-07/fraud-augmentation-pipeline.git
cd fraud-augmentation-pipeline
```

### Step 2: Prepare the Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Download Dataset
Download `creditcard.csv` from Kaggle and place it in `data/raw/`.

### Step 4: Train CTGAN Model
```bash
python augmentation/train_ctgan.py
```
This generates `data/models/ctgan_model.pkl`.

### Step 5: Start the Pipeline
```bash
docker compose up --build
```
Wait for all services to initialize.

### Step 6: View the Dashboard
Open [http://localhost:8501](http://localhost:8501) in your browser.

### Step 7: Evaluate Models
```bash
docker compose exec stream-processor python models/train_eval.py
```

### Step 8: Stop the Pipeline
```bash
docker compose down -v
rm -rf tmp/spark_checkpoint
```

---

## 📊 Results

| Classifier          | Baseline F1 | SMOTE F1 | CTGAN F1 |
|---------------------|-------------|----------|----------|
| Random Forest       | 0.732       | 0.798    | 0.841    |
| XGBoost             | 0.751       | 0.812    | 0.857    |
| Logistic Regression | 0.689       | 0.743    | 0.779    |

CTGAN consistently achieves the best minority-class F1 score.

---

## 🛠️ Troubleshooting

- **Offset mismatch error**: Run `docker compose down -v && rm -rf tmp/spark_checkpoint && docker compose up --build`.
- **Synthetic rows = 0**: Ensure `data/models/ctgan_model.pkl` exists before starting Docker.
- **Dashboard not refreshing**: Check Streamlit logs for errors.
- **Model validation empty**: Ensure the pipeline has been running for at least 5 minutes.

---

## 📚 Acknowledgments

This project is based on the paper: *Generative AI for Real-Time Data Augmentation in Big Data Pipelines, CE2CT 2025*.
