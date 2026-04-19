import pandas as pd
import yaml
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation.metrics import evaluate_model, save_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("train_eval")

def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_schema():
    with open("configs/schema.json", "r") as f:
        return json.load(f)

def load_dataset(dir_path):
    files = list(Path(dir_path).glob("*.parquet"))
    if not files:
        label = "sample implementation" if 'sample' in str(dir_path) else str(dir_path)
        log.warning(f"No parquet files found in {dir_path}. Ensure pipeline has written data.")
        return None
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

def train_and_evaluate(df, target_col, dataset_name, test_size, random_state, classifiers_list):
    log.info(f"Training on {dataset_name} dataset... (Total rows: {len(df)})")
    
    # Exclude metadata columns from features
    exclude_cols = [target_col, "is_synthetic", "batch_id"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    all_metrics = []
    
    for clf_name in classifiers_list:
        log.info(f"Training {clf_name} on {dataset_name}...")
        if clf_name == "random_forest":
            clf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        elif clf_name == "xgboost":
            clf = XGBClassifier(random_state=random_state, eval_metric="logloss")
        elif clf_name == "logistic_regression":
            clf = LogisticRegression(random_state=random_state, max_iter=1000)
        else:
            log.warning(f"Unknown classifier: {clf_name}")
            continue
            
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        y_prob = None
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)[:, 1]
            
        metrics = evaluate_model(y_test, y_pred, y_prob, clf_name, dataset_name)
        all_metrics.append(metrics)
        log.info(f"{clf_name} completed. F1 Macro: {metrics['f1_macro']:.4f}, Minority F1: {metrics['f1_minority']:.4f}")
        
    return all_metrics

def main():
    config = load_config()
    schema = load_schema()
    
    target_col = schema["target_column"]
    test_size = config["model"]["test_size"]
    random_state = config["model"]["random_state"]
    classifiers = config["model"]["classifiers"]
    
    processed_dir = config["paths"]["processed_dir"]
    augmented_dir = config["paths"]["augmented_dir"]
    
    # Train on Baseline
    df_base = load_dataset(processed_dir)
    all_results = []
    if df_base is not None:
        metrics_base = train_and_evaluate(df_base, target_col, "baseline", test_size, random_state, classifiers)
        all_results.extend(metrics_base)
        
    # Train on Augmented
    df_aug = load_dataset(augmented_dir)
    if df_aug is not None:
        metrics_aug = train_and_evaluate(df_aug, target_col, "augmented", test_size, random_state, classifiers)
        all_results.extend(metrics_aug)
        
    if all_results:
        save_metrics(all_results, output_dir=config["paths"]["results_dir"])
        log.info("All metrics saved successfully.")
    else:
        log.warning("No models were trained due to missing data.")

if __name__ == "__main__":
    main()
