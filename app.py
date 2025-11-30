import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss

from train import train_and_save, Paths, add_engineered_features


st.set_page_config(page_title="Admissions Predictor", page_icon="🎓", layout="wide")
st.title("🎓 Admissions Probability Predictor")


PATHS = Paths()
MODEL_PATH = PATHS.model_path


@st.cache_resource
def load_or_train_model():
    if not MODEL_PATH.exists():
        bundle = train_and_save(PATHS, seed=42)
        return bundle
    return joblib.load(MODEL_PATH)


bundle = load_or_train_model()
model = bundle["model"]
THRESHOLD = float(bundle.get("threshold", 0.5))
FEATURE_COLS = bundle.get("feature_columns", None)


def align_to_training_columns(df: pd.DataFrame, feature_cols) -> pd.DataFrame:
    if feature_cols is None:
        return df
    out = df.copy()
    for c in feature_cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[feature_cols]


def predict_proba_df(df_in: pd.DataFrame) -> np.ndarray:
    df_feat = add_engineered_features(df_in)
    X = align_to_training_columns(df_feat, FEATURE_COLS)
    p = model.predict_proba(X)[:, 1]
    return np.clip(p, 1e-15, 1 - 1e-15)


# Sidebar
st.sidebar.header("Model Info")
st.sidebar.write(f"**Decision threshold:** `{THRESHOLD:.3f}`")
st.sidebar.write(f"**Model file:** `{MODEL_PATH.as_posix()}`")
if "meta" in bundle:
    st.sidebar.write(f"**Best model:** `{bundle['meta'].get('best_model')}`")

if st.sidebar.button("Retrain model (fresh)"):
    # clear cache + retrain
    load_or_train_model.clear()
    bundle = load_or_train_model()
    st.rerun()


tab1, tab2, tab3 = st.tabs(["🧍 Single Prediction", "📄 Batch Scoring", "📊 Evaluate (optional)"])


# -----------------------------
# Tab 1: Single Prediction
# -----------------------------
with tab1:
    st.subheader("Single Prediction (Manual Input)")

    c1, c2, c3 = st.columns(3)

    with c1:
        term = st.selectbox("term", ["Fall 2024", "Spring 2025", "Fall 2025", "Spring 2026"])
        program = st.selectbox("program", ["MS", "MTech", "MBA", "PhD"])
        major = st.selectbox("major", ["CS", "DS", "ECE", "Mech", "Civil", "Bio", "Econ", "Finance"])
        country = st.text_input("country", value="India")
        uni_tier = st.selectbox("uni_tier", ["Tier1", "Tier2", "Tier3"])

    with c2:
        cgpa_10 = st.number_input("cgpa_10", 5.5, 10.0, 7.8, 0.01)
        gre_total = st.number_input("gre_total", 260, 340, 310, 1)
        gre_aw = st.number_input("gre_aw", 0.0, 6.0, 4.0, 0.5)
        toefl = st.number_input("toefl", 60, 120, 95, 1)
        ielts = st.number_input("ielts (optional)", 0.0, 9.0, 7.0, 0.5)

    with c3:
        work_exp_years = st.number_input("work_exp_years", 0.0, 6.0, 1.5, 0.1)
        internships = st.number_input("internships", 0, 4, 1, 1)
        projects = st.number_input("projects", 0, 8, 3, 1)
        publications = st.number_input("publications", 0, 5, 0, 1)
        research_exp = st.checkbox("research_exp", value=False)

        sop_score = st.number_input("sop_score", 1.0, 5.0, 3.5, 0.5)
        lor_score = st.number_input("lor_score", 1.0, 5.0, 3.5, 0.5)
        portfolio_score = st.number_input("portfolio_score", 1.0, 5.0, 3.0, 0.5)
        interview_score = st.number_input("interview_score", 1.0, 5.0, 3.0, 0.5)

        gender = st.selectbox("gender", ["Female", "Male", "Non-binary"])
        income_bucket = st.selectbox("income_bucket", ["Low", "Middle", "High"])
        first_gen = st.checkbox("first_gen", value=False)

    row = pd.DataFrame([{
        "term": term, "program": program, "major": major, "country": country, "uni_tier": uni_tier,
        "cgpa_10": cgpa_10, "gre_total": gre_total, "gre_aw": gre_aw, "toefl": toefl, "ielts": ielts,
        "work_exp_years": work_exp_years, "internships": internships, "projects": projects,
        "research_exp": research_exp, "publications": publications,
        "sop_score": sop_score, "lor_score": lor_score, "portfolio_score": portfolio_score, "interview_score": interview_score,
        "gender": gender, "income_bucket": income_bucket, "first_gen": first_gen
    }])

    if st.button("Predict"):
        p = float(predict_proba_df(row)[0])
        decision = "✅ Admit" if p >= THRESHOLD else "❌ Reject"
        st.metric("Predicted probability", f"{p:.3f}")
        st.write(f"**Decision (@ threshold {THRESHOLD:.3f}):** {decision}")


# -----------------------------
# Tab 2: Batch scoring
# -----------------------------
with tab2:
    st.subheader("Batch Scoring (Upload CSV → Download Scored CSV)")

    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        df_up = pd.read_csv(up)
        st.write("Preview:")
        st.dataframe(df_up.head(10))

        p = predict_proba_df(df_up)
        scored = df_up.copy()
        scored["pred_prob"] = np.round(p, 6)
        scored["pred_label"] = (p >= THRESHOLD).astype(int)

        st.write("Scored preview:")
        st.dataframe(scored.head(10))

        buffer = io.StringIO()
        scored.to_csv(buffer, index=False)
        st.download_button(
            "⬇️ Download scored CSV",
            data=buffer.getvalue(),
            file_name="scored_predictions.csv",
            mime="text/csv"
        )


# -----------------------------
# Tab 3: Evaluate (optional)
# -----------------------------
with tab3:
    st.subheader("Evaluate Model (optional)")

    st.write("Upload a CSV that includes `admit_label` to compute metrics.")
    up_eval = st.file_uploader("Upload labeled CSV", type=["csv"], key="eval")
    if up_eval is not None:
        df_eval = pd.read_csv(up_eval)
        if "admit_label" not in df_eval.columns:
            st.error("This CSV does not contain `admit_label`.")
        else:
            y_true = df_eval["admit_label"].astype(int).values
            p = predict_proba_df(df_eval)

            ll = log_loss(y_true, np.clip(p, 1e-15, 1 - 1e-15))
            bs = brier_score_loss(y_true, p)
            auc = roc_auc_score(y_true, p) if len(np.unique(y_true)) == 2 else np.nan
            pr = average_precision_score(y_true, p) if len(np.unique(y_true)) == 2 else np.nan

            st.write({
                "roc_auc": None if np.isnan(auc) else float(auc),
                "pr_auc": None if np.isnan(pr) else float(pr),
                "logloss": float(ll),
                "brier": float(bs),
                "threshold": float(THRESHOLD),
            })

            y_pred = (p >= THRESHOLD).astype(int)
            st.write(f"**Predicted admit rate (@ threshold):** {float(y_pred.mean()):.3f}")
