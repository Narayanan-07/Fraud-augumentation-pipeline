"""
Streamlit Dashboard — Person 2 owns and implements this file.
Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Fraud Augmentation Pipeline", layout="wide")
st.title("Fraud Augmentation Pipeline — Dashboard")

st.info("Person 2: implement this dashboard in dashboard/app.py")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Processed Batches", len(list(Path("data/processed").glob("*.parquet"))))
with col2:
    st.metric("Augmented Batches", len(list(Path("data/augmented").glob("*.parquet"))))
with col3:
    st.metric("Status", "Running" if Path("tmp/spark_checkpoint").exists() else "Not started")

st.subheader("Augmented Data Preview")
aug_files = list(Path("data/augmented").glob("*.parquet"))
if aug_files:
    df = pd.read_parquet(sorted(aug_files)[-1])
    st.dataframe(df.head(20))
    st.write(f"Synthetic rows: {df['is_synthetic'].sum()} / {len(df)}")
else:
    st.warning("No augmented batches yet. Run the pipeline first.")
