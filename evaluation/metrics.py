import json
import os
import logging
from sklearn.metrics import classification_report, f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score
import pandas as pd

log = logging.getLogger(__name__)

def evaluate_model(y_true, y_pred, y_prob, model_name, data_source):
    """
    Evaluates model predictions and returns a dictionary of metrics.
    """
    f1_mac = f1_score(y_true, y_pred, average='macro')
    # Assuming fraud is label 1
    f1_min = f1_score(y_true, y_pred, pos_label=1)
    prec = precision_score(y_true, y_pred, pos_label=1)
    rec = recall_score(y_true, y_pred, pos_label=1)
    
    roc_auc = None
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            pass
            
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    metrics = {
        "model": model_name,
        "data_source": data_source, # e.g., 'baseline', 'smote', 'ctgan'
        "f1_macro": f1_mac,
        "f1_minority": f1_min,
        "precision": prec,
        "recall": rec,
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }
    
    return metrics

def save_metrics(metrics_list, output_dir="evaluation/results"):
    """
    Saves the list of metrics to a JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "metrics.json")
    
    # Load existing if any, so we can append/update
    existing_metrics = []
    if os.path.exists(out_file):
        try:
            with open(out_file, "r") as f:
                existing_metrics = json.load(f)
        except json.JSONDecodeError:
            pass
            
    # We overwrite metrics for the same model + data_source combination
    for m in metrics_list:
        existing_metrics = [x for x in existing_metrics if not (x["model"] == m["model"] and x["data_source"] == m["data_source"])]
        existing_metrics.append(m)
        
    with open(out_file, "w") as f:
        json.dump(existing_metrics, f, indent=4)
        
    log.info(f"Metrics saved to {out_file}")

def get_latest_metrics(output_dir="evaluation/results"):
    out_file = os.path.join(output_dir, "metrics.json")
    if os.path.exists(out_file):
        with open(out_file, "r") as f:
            return json.load(f)
    return []
