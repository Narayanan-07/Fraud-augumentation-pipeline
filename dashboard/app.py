# dashboard/app.py  — Fraud Augmentation Pipeline Dashboard
import time, json, yaml, subprocess, os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Fraud Augmentation Pipeline", layout="wide", page_icon="🛡️")

# ── Auto-refresh (Bug 5 fix) ─────────────────────────────────────────────────
count = st_autorefresh(interval=5000, key="autorefresh")

# ── Config ───────────────────────────────────────────────────────────────────
try:
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
except Exception:
    config = {"paths": {"processed_dir": "data/processed", "augmented_dir": "data/augmented", "results_dir": "evaluation/results"}}

PROCESSED_DIR = Path(config["paths"]["processed_dir"])
AUGMENTED_DIR = Path(config["paths"]["augmented_dir"])
RESULTS_FILE  = Path(config["paths"]["results_dir"]) / "metrics.json"

# ── Cached loaders ───────────────────────────────────────────────────────────
@st.cache_data(ttl=5)
def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

@st.cache_data(ttl=5)
def get_aug_files():
    return sorted(AUGMENTED_DIR.glob("*.parquet"))

@st.cache_data(ttl=5)
def get_proc_files():
    return sorted(PROCESSED_DIR.glob("*.parquet"))

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Fraud Pipeline")
    st.markdown(f"🕐 `{pd.Timestamp.now().strftime('%H:%M:%S')}`")
    st.divider()
    st.markdown("### Pipeline Layers")
    layers = [("🔵","Ingestion (Kafka)"),("🔵","Processing (Spark)"),
              ("🟢","Augmentation (CTGAN)"),("🟡","Training (ML)"),("🟢","Dashboard")]
    for icon, name in layers:
        st.markdown(f"{icon} {name}")
    st.divider()
    if st.button("🔄 Manual Refresh"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Auto-refreshing every 5s")
    if st.button("▶️ Run Evaluation"):
        with st.spinner("Running train_eval.py..."):
            r = subprocess.run(["python","models/train_eval.py"],
                               capture_output=True, text=True, timeout=300)
            st.code(r.stdout[-2000:] if r.stdout else r.stderr[-2000:])

# ── Title ────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='color:#00ff88'>🛡️ Fraud Augmentation Pipeline</h1>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📡 Live Monitor","🔬 Data Quality","📊 Model Validation","⚙️ System Info"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Pipeline Monitor
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    try:
        proc_files = get_proc_files()
        aug_files  = get_aug_files()

        # KPI row
        k1,k2,k3,k4 = st.columns(4)
        total_real, total_synth = 0, 0
        if aug_files:
            try:
                last_df = load_parquet(str(aug_files[-1]))
                total_real  = int((~last_df["is_synthetic"]).sum()) if "is_synthetic" in last_df.columns else len(last_df)
                total_synth = int(last_df["is_synthetic"].sum()) if "is_synthetic" in last_df.columns else 0
            except Exception:
                pass

        k1.metric("Processed Batches", len(proc_files))
        k2.metric("Augmented Batches", len(aug_files))
        k3.metric("Total Real Records", f"{total_real:,}")
        k4.metric("Total Synthetic Records", f"{total_synth:,}")

        # Pipeline status
        if aug_files:
            age = time.time() - aug_files[-1].stat().st_mtime
            if age < 30:
                st.success("🟢 RUNNING — last write < 30s ago")
            elif age < 120:
                st.warning(f"🟡 STALLED — last write {int(age)}s ago")
            else:
                st.error(f"🔴 OFFLINE — last write {int(age)}s ago")
        else:
            st.error("🔴 OFFLINE — no augmented files found")

        # Real-time batch chart
        if aug_files:
            recent = aug_files[-20:]
            rows = []
            for f in recent:
                try:
                    df = load_parquet(str(f))
                    rows.append({
                        "batch": f.stem,
                        "Real": int((~df["is_synthetic"]).sum()) if "is_synthetic" in df.columns else len(df),
                        "Synthetic": int(df["is_synthetic"].sum()) if "is_synthetic" in df.columns else 0,
                        "mtime": f.stat().st_mtime,
                    })
                except Exception:
                    pass
            if rows:
                cdf = pd.DataFrame(rows)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=cdf["batch"], y=cdf["Real"], name="Real Rows",
                                         line=dict(color="#4da6ff",width=2), mode="lines+markers"))
                fig.add_trace(go.Scatter(x=cdf["batch"], y=cdf["Synthetic"], name="Synthetic Rows",
                                         line=dict(color="#00ff88",width=2), mode="lines+markers"))
                fig.update_layout(template="plotly_dark", title="Last 20 Batches — Row Counts",
                                  xaxis_title="Batch", yaxis_title="Rows", height=350)
                st.plotly_chart(fig, use_container_width=True)

                # Throughput gauge
                if len(rows) >= 2:
                    dt = rows[-1]["mtime"] - rows[0]["mtime"]
                    total_rows = sum(r["Real"]+r["Synthetic"] for r in rows)
                    rps = total_rows / max(dt, 1)
                    fig2 = go.Figure(go.Indicator(mode="gauge+number",
                        value=round(rps,2), title={"text":"Records / sec"},
                        gauge={"axis":{"range":[0,500]},
                               "bar":{"color":"#00ff88"},
                               "steps":[{"range":[0,100],"color":"#1a1f2e"},
                                        {"range":[100,300],"color":"#223344"},
                                        {"range":[300,500],"color":"#1a3322"}]}))
                    fig2.update_layout(template="plotly_dark", height=250)
                    st.plotly_chart(fig2, use_container_width=True)

                # Batch latency sparkline
                mtimes = [r["mtime"] for r in rows]
                gaps = [round(mtimes[i]-mtimes[i-1],2) for i in range(1,len(mtimes))]
                if gaps:
                    fig3 = go.Figure(go.Scatter(y=gaps, mode="lines+markers",
                        line=dict(color="#f0a500",width=1.5), name="Latency (s)"))
                    fig3.update_layout(template="plotly_dark", title="Batch Latency (s)",
                                       yaxis_title="Seconds", height=200)
                    st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:
        st.error(f"Tab 1 error: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Data Quality & Augmentation
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    try:
        aug_files = get_aug_files()
        if not aug_files:
            st.warning("No augmented batches yet. Start the pipeline first.")
        else:
            df_aug = load_parquet(str(aug_files[-1]))
            has_syn = "is_synthetic" in df_aug.columns

            # Class distribution stacked bar
            if "Class" in df_aug.columns and has_syn:
                dist = df_aug.groupby(["Class","is_synthetic"]).size().reset_index(name="count")
                dist["Type"] = dist["is_synthetic"].map({True:"Synthetic",False:"Real"})
                dist["Class"] = dist["Class"].map({0:"Normal",1:"Fraud"})
                fig = px.bar(dist, x="Class", y="count", color="Type",
                             color_discrete_map={"Real":"#4da6ff","Synthetic":"#00ff88"},
                             title="Class Distribution (Real vs Synthetic)",
                             template="plotly_dark", barmode="stack")
                st.plotly_chart(fig, use_container_width=True)

            # Augmentation ratio trend
            recent = aug_files[-20:]
            ratios = []
            zero_synth_streak = 0
            for f in recent:
                try:
                    df = load_parquet(str(f))
                    syn = int(df["is_synthetic"].sum()) if "is_synthetic" in df.columns else 0
                    tot = len(df)
                    ratios.append({"batch": f.stem, "ratio": syn/tot if tot>0 else 0, "synthetic": syn})
                    if syn == 0:
                        zero_synth_streak += 1
                    else:
                        zero_synth_streak = 0
                except Exception:
                    pass

            # Bug 2: warn if 5+ consecutive zero-synthetic batches
            if zero_synth_streak >= 5:
                st.error(f"⚠️ **Synthetic rows = 0 for {zero_synth_streak} consecutive batches!**\n\n"
                         "**Fix:** Set `min_samples_to_augment: 1` in `configs/config.yaml` "
                         "and ensure `data/models/ctgan_model.pkl` exists. "
                         "Run: `docker compose exec stream-processor python augmentation/train_ctgan.py`")

            if ratios:
                rdf = pd.DataFrame(ratios)
                fig2 = go.Figure(go.Scatter(x=rdf["batch"], y=rdf["ratio"], mode="lines+markers",
                    fill="tozeroy", line=dict(color="#00ff88"), name="Synth Ratio"))
                fig2.update_layout(template="plotly_dark", title="Augmentation Ratio (last 20 batches)",
                                   yaxis_title="Synthetic / Total", height=300)
                st.plotly_chart(fig2, use_container_width=True)

            # Feature distribution table
            v_cols = [c for c in [f"V{i}" for i in range(1,11)] + ["Amount_scaled"] if c in df_aug.columns]
            if v_cols and has_syn:
                real_df = df_aug[~df_aug["is_synthetic"]]
                syn_df  = df_aug[df_aug["is_synthetic"]]
                stats = []
                for c in v_cols:
                    stats.append({"Feature":c,
                                  "Real Mean":round(real_df[c].mean(),4) if len(real_df)>0 else "—",
                                  "Real Std":round(real_df[c].std(),4) if len(real_df)>0 else "—",
                                  "Synth Mean":round(syn_df[c].mean(),4) if len(syn_df)>0 else "—",
                                  "Synth Std":round(syn_df[c].std(),4) if len(syn_df)>0 else "—"})
                st.subheader("Feature Distribution: Real vs Synthetic")
                st.dataframe(pd.DataFrame(stats), use_container_width=True)

            # Raw preview with highlighted synthetic rows
            st.subheader("Latest Batch Preview (last 20 rows)")
            preview = df_aug.tail(20).copy()
            if has_syn:
                def highlight_syn(row):
                    if row.get("is_synthetic", False):
                        return ["background-color: #0d2b1a; color:#00ff88"]*len(row)
                    return [""]*len(row)
                st.dataframe(preview.style.apply(highlight_syn, axis=1), use_container_width=True)
            else:
                st.dataframe(preview, use_container_width=True)
    except Exception as e:
        st.error(f"Tab 2 error: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Validation
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    try:
        if not RESULTS_FILE.exists():
            st.warning("No metrics.json found.")
            if st.button("▶️ Run Evaluation Now", key="eval_btn"):
                with st.spinner("Running models/train_eval.py ..."):
                    r = subprocess.run(["python","models/train_eval.py"],
                                       capture_output=True, text=True, timeout=300)
                    st.code(r.stdout[-3000:] if r.stdout else r.stderr[-3000:])
                st.rerun()
        else:
            with open(RESULTS_FILE) as f:
                metrics_data = json.load(f)

            if not metrics_data:
                st.info("metrics.json is empty. Run evaluation first.")
            else:
                mdf = pd.DataFrame(metrics_data)

                # KPI row
                best_f1  = mdf["f1_minority"].max() if "f1_minority" in mdf.columns else 0
                best_auc = mdf["roc_auc"].max() if "roc_auc" in mdf.columns else 0
                best_row = mdf.loc[mdf["f1_minority"].idxmax()] if "f1_minority" in mdf.columns else mdf.iloc[0]
                baseline = mdf[mdf["data_source"]=="baseline"]["f1_minority"].max() if "baseline" in mdf.get("data_source","").values else 0

                k1,k2,k3,k4 = st.columns(4)
                k1.metric("Best F1-Minority", f"{best_f1:.3f}")
                k2.metric("Best ROC-AUC", f"{best_auc:.3f}")
                k3.metric("Best Model", str(best_row.get("model","")))
                imp = best_f1 - baseline
                k4.metric("Improvement over Baseline", f"+{imp:.3f}" if imp>0 else f"{imp:.3f}")

                color_map = {"baseline":"#ff4444","smote":"#ff9900","ctgan":"#00ff88","augmented":"#4da6ff"}

                # F1-Minority grouped bar
                if "f1_minority" in mdf.columns:
                    fig = go.Figure()
                    for src in mdf["data_source"].unique():
                        sub = mdf[mdf["data_source"]==src]
                        fig.add_trace(go.Bar(x=sub["model"], y=sub["f1_minority"],
                            name=src, marker_color=color_map.get(src,"#888888")))
                    fig.update_layout(template="plotly_dark", barmode="group",
                        title="F1-Minority by Model & Data Source", height=350)
                    st.plotly_chart(fig, use_container_width=True)

                # ROC-AUC grouped bar
                if "roc_auc" in mdf.columns:
                    fig2 = go.Figure()
                    for src in mdf["data_source"].unique():
                        sub = mdf[mdf["data_source"]==src]
                        fig2.add_trace(go.Bar(x=sub["model"], y=sub["roc_auc"],
                            name=src, marker_color=color_map.get(src,"#888888")))
                    fig2.update_layout(template="plotly_dark", barmode="group",
                        title="ROC-AUC by Model & Data Source", height=350)
                    st.plotly_chart(fig2, use_container_width=True)

                # Full metrics table
                st.subheader("Full Metrics Table")
                num_cols = [c for c in ["f1_minority","f1_macro","roc_auc","precision","recall"] if c in mdf.columns]
                display = mdf.drop(columns=["confusion_matrix"], errors="ignore")
                try:
                    styled = display.style.highlight_max(subset=num_cols, color="#0d3320")\
                                          .highlight_min(subset=num_cols, color="#3d0d0d")
                    st.dataframe(styled, use_container_width=True)
                except Exception:
                    st.dataframe(display, use_container_width=True)

                # Confusion matrices
                st.subheader("Confusion Matrices")
                cols_cm = st.columns(min(len(metrics_data), 3))
                for i, m in enumerate(metrics_data):
                    if "confusion_matrix" in m:
                        with cols_cm[i % 3]:
                            cm = m["confusion_matrix"]
                            fig3 = px.imshow([[cm[0][0],cm[0][1]],[cm[1][0],cm[1][1]]],
                                text_auto=True, color_continuous_scale="Viridis",
                                x=["Pred Normal","Pred Fraud"], y=["Act Normal","Act Fraud"],
                                title=f"{m['model']}<br>{m.get('data_source','')}",
                                template="plotly_dark")
                            fig3.update_layout(height=250, coloraxis_showscale=False)
                            st.plotly_chart(fig3, use_container_width=True)

                # Augmentation impact summary
                st.subheader("Augmentation Impact Summary")
                summary_lines = []
                for model in mdf["model"].unique():
                    mrows = mdf[mdf["model"]==model]
                    base = mrows[mrows["data_source"]=="baseline"]
                    for src in ["ctgan","smote","augmented"]:
                        aug = mrows[mrows["data_source"]==src]
                        if not base.empty and not aug.empty and "f1_minority" in mdf.columns:
                            delta = aug["f1_minority"].values[0] - base["f1_minority"].values[0]
                            direction = "improved" if delta>0 else "decreased"
                            summary_lines.append(
                                f"• **{src.upper()}** {direction} minority F1 by "
                                f"**{abs(delta)*100:.1f} pp** over baseline for **{model}**."
                            )
                if summary_lines:
                    st.markdown("\n".join(summary_lines))
                else:
                    st.info("Need both baseline and augmented results to compute impact.")
    except Exception as e:
        st.error(f"Tab 3 error: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — System Info
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    try:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🐳 Container Info")
            st.code(f"Hostname : {os.environ.get('HOSTNAME', os.popen('hostname').read().strip())}\n"
                    f"KAFKA    : {os.environ.get('KAFKA_BROKER','kafka:9092')}\n"
                    f"ENV      : {os.environ.get('PIPELINE_ENV','local')}")

            st.subheader("📁 File System Stats")
            for label, d in [("Processed", PROCESSED_DIR), ("Augmented", AUGMENTED_DIR),
                              ("Results", RESULTS_FILE.parent)]:
                if d.exists():
                    files = list(d.glob("*"))
                    sz = sum(f.stat().st_size for f in files if f.is_file())
                    st.markdown(f"**{label}**: {len([f for f in files if f.is_file()])} files, "
                                f"{sz/1e6:.2f} MB")
                else:
                    st.markdown(f"**{label}**: _directory not found_")

        with c2:
            st.subheader("⚙️ Config")
            try:
                with open("configs/config.yaml") as f:
                    st.code(f.read(), language="yaml")
            except Exception:
                st.warning("configs/config.yaml not found")

            st.subheader("📐 Schema")
            try:
                with open("configs/schema.json") as f:
                    st.code(f.read(), language="json")
            except Exception:
                st.warning("configs/schema.json not found")

        st.subheader("📋 Stream Processor Logs (last 50 lines)")
        log_path = Path("logs/stream_processor.log")
        if log_path.exists():
            lines = log_path.read_text(errors="replace").splitlines()
            st.code("\n".join(lines[-50:]))
        else:
            st.info("No log file at logs/stream_processor.log")
    except Exception as e:
        st.error(f"Tab 4 error: {e}")