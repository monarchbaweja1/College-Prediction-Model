import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss


# -----------------------------
# Paths (works locally + Streamlit Cloud)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACTS_DIR / "admissions_model.pkl"


# -----------------------------
# Utilities
# -----------------------------
def _clip(a, lo, hi):
    return np.clip(a, lo, hi)


def make_ohe():
    # sklearn compatibility: sparse_output (new) vs sparse (old)
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # IELTS fill if present
    if "ielts" in out.columns:
        out["ielts"] = pd.to_numeric(out["ielts"], errors="coerce").fillna(7.0)

    # Ensure numeric types
    for col in [
        "gre_total","toefl","cgpa_10","gre_aw","work_exp_years",
        "internships","projects","publications",
        "sop_score","lor_score","portfolio_score","interview_score"
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Derived features if base cols exist
    needed = ["gre_total", "toefl", "cgpa_10", "gre_aw"]
    if all(c in out.columns for c in needed):
        out["gre_norm"]   = (out["gre_total"] - 260) / 80
        out["toefl_norm"] = (out["toefl"] - 60) / 60
        out["cgpa_norm"]  = (out["cgpa_10"] - 5.5) / 4.5
        out["aw_norm"]    = out["gre_aw"] / 6.0

        out["academic_index"] = 0.55 * out["cgpa_norm"] + 0.45 * out["gre_norm"]

        if "ielts" in out.columns:
            ielts_norm = out["ielts"] / 9.0
        else:
            ielts_norm = 7.0 / 9.0

        out["english_index"] = 0.70 * out["toefl_norm"] + 0.30 * ielts_norm

    needed_profile = [
        "work_exp_years","internships","projects","publications",
        "sop_score","lor_score","portfolio_score","interview_score"
    ]
    if all(c in out.columns for c in needed_profile):
        out["profile_index"] = (
            0.20 * (out["work_exp_years"] / 6.0)
            + 0.10 * (out["internships"] / 4.0)
            + 0.10 * (out["projects"] / 8.0)
            + 0.15 * (out["publications"].clip(0, 5) / 5.0)
            + 0.15 * ((out["sop_score"] - 1.0) / 4.0)
            + 0.15 * ((out["lor_score"] - 1.0) / 4.0)
            + 0.10 * ((out["portfolio_score"] - 1.0) / 4.0)
            + 0.05 * ((out["interview_score"] - 1.0) / 4.0)
        )

    if "cgpa_norm" in out.columns:
        out["cgpa_sq"] = out["cgpa_norm"] ** 2

    if "gre_norm" in out.columns and "cgpa_norm" in out.columns:
        out["gre_x_cgpa"] = out["gre_norm"] * out["cgpa_norm"]

    if "sop_score" in out.columns and "lor_score" in out.columns:
        out["sop_x_lor"] = ((out["sop_score"] - 1) / 4) * ((out["lor_score"] - 1) / 4)

    if "gre_norm" in out.columns and "toefl_norm" in out.columns:
        out["test_strength"] = 0.6 * out["gre_norm"] + 0.4 * out["toefl_norm"]

    return out


def generate_dataset(n=21534, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    terms = ["Fall 2024", "Spring 2025", "Fall 2025", "Spring 2026"]
    programs = ["MS", "MTech", "MBA", "PhD"]
    majors = ["CS", "DS", "ECE", "Mech", "Civil", "Bio", "Econ", "Finance"]
    tiers = ["Tier1", "Tier2", "Tier3"]
    tier_probs = [0.25, 0.45, 0.30]

    countries = [
        "India","United States","Canada","United Kingdom","Germany","France","Netherlands",
        "Australia","New Zealand","Singapore","UAE","Saudi Arabia","Qatar",
        "Brazil","Mexico","South Africa","Nigeria","Egypt",
        "China","Japan","South Korea","Vietnam","Thailand","Indonesia","Malaysia",
        "Pakistan","Bangladesh","Sri Lanka","Turkey","Russia","Poland","Spain","Italy"
    ]

    genders = ["Female", "Male", "Non-binary"]
    gender_probs = [0.45, 0.53, 0.02]

    income_buckets = ["Low", "Middle", "High"]
    income_probs = [0.35, 0.50, 0.15]

    cgpa = _clip(rng.normal(7.6, 0.9, n), 5.5, 10.0)
    gre = _clip((260 + (cgpa - 5.5) / 4.5 * 80 + rng.normal(0, 10, n)).round(), 260, 340).astype(int)
    gre_aw = _clip((rng.normal(4.0, 0.9, n) + (cgpa - 7.5) * 0.15), 0, 6)
    gre_aw = (np.round(gre_aw * 2) / 2).astype(float)

    toefl = _clip((rng.normal(95, 12, n) + (gre - 300) * 0.10).round(), 60, 120).astype(int)

    ielts = _clip(rng.normal(7.2, 0.7, n), 5.0, 9.0)
    ielts = (np.round(ielts * 2) / 2).astype(float)
    ielts_mask = rng.random(n) < 0.40
    ielts[ielts_mask] = np.nan

    work_exp = _clip(rng.normal(1.8, 1.7, n), 0, 6)
    work_exp = np.round(work_exp, 1)

    internships = _clip(rng.poisson(1.2, n), 0, 4).astype(int)
    projects = _clip(rng.poisson(3.0, n), 0, 8).astype(int)

    research_exp = (rng.random(n) < _clip(0.18 + (cgpa - 7.5) * 0.04, 0.05, 0.55)).astype(bool)
    publications = _clip(rng.poisson(0.3 + research_exp.astype(int) * 0.7, n), 0, 5).astype(int)

    def score_1_5(mean, sd):
        s = _clip(rng.normal(mean, sd, n), 1, 5)
        return np.round(s * 2) / 2

    sop = score_1_5(3.4 + (cgpa - 7.5) * 0.12, 0.75)
    lor = score_1_5(3.3 + (work_exp - 1.5) * 0.10, 0.70)
    portfolio = score_1_5(3.1 + (projects - 3.0) * 0.07, 0.75)
    interview = score_1_5(3.2 + (gre - 300) * 0.01, 0.85)

    df = pd.DataFrame({
        "term": rng.choice(terms, n),
        "program": rng.choice(programs, n, p=[0.45, 0.25, 0.22, 0.08]),
        "major": rng.choice(majors, n, p=[0.25, 0.20, 0.15, 0.10, 0.08, 0.07, 0.08, 0.07]),
        "country": rng.choice(countries, n),
        "uni_tier": rng.choice(tiers, n, p=tier_probs),

        "cgpa_10": np.round(cgpa, 2),
        "gre_total": gre,
        "gre_aw": gre_aw,
        "toefl": toefl,
        "ielts": ielts,

        "work_exp_years": work_exp,
        "internships": internships,
        "projects": projects,
        "research_exp": research_exp,
        "publications": publications,

        "sop_score": sop,
        "lor_score": lor,
        "portfolio_score": portfolio,
        "interview_score": interview,

        "gender": rng.choice(genders, n, p=gender_probs),
        "income_bucket": rng.choice(income_buckets, n, p=income_probs),
        "first_gen": (rng.random(n) < 0.38)
    })

    # Create targets
    ielts_filled = df["ielts"].fillna(7.0)
    gre_norm = (df["gre_total"] - 260) / 80
    toefl_norm = (df["toefl"] - 60) / 60
    cgpa_norm = (df["cgpa_10"] - 5.5) / 4.5

    english_index = 0.70 * toefl_norm + 0.30 * (ielts_filled / 9.0)
    academic_index = 0.55 * cgpa_norm + 0.45 * gre_norm

    profile_index = (
        0.20 * (df["work_exp_years"] / 6.0)
        + 0.10 * (df["internships"] / 4.0)
        + 0.10 * (df["projects"] / 8.0)
        + 0.15 * (df["publications"].clip(0, 5) / 5.0)
        + 0.15 * ((df["sop_score"] - 1.0) / 4.0)
        + 0.15 * ((df["lor_score"] - 1.0) / 4.0)
        + 0.10 * ((df["portfolio_score"] - 1.0) / 4.0)
        + 0.05 * ((df["interview_score"] - 1.0) / 4.0)
    )

    tier_boost = df["uni_tier"].map({"Tier1": 0.08, "Tier2": 0.03, "Tier3": -0.02}).astype(float)
    research_boost = df["research_exp"].astype(int) * 0.08

    z = (
        -1.2
        + 2.4 * academic_index
        + 1.1 * profile_index
        + 0.6 * english_index
        + tier_boost
        + research_boost
        + 0.25 * (gre_norm * cgpa_norm)
        + 0.15 * (((df["sop_score"] - 1) / 4) * ((df["lor_score"] - 1) / 4))
        + rng.normal(0.0, 0.25, size=len(df))
    )

    chance = 1 / (1 + np.exp(-z))
    chance = np.clip(chance, 0.01, 0.99)
    admit = rng.binomial(1, chance, size=len(df)).astype(int)

    df["admit_label"] = admit
    return df


def find_best_threshold(y_true: np.ndarray, p: np.ndarray):
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_f1 = 0.5, -1.0

    for t in thresholds:
        pred = (p >= t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)

    return best_t, best_f1


def train_and_save(seed=42):
    df = generate_dataset(n=21534, seed=seed)
    df = add_engineered_features(df)

    y = df["admit_label"].astype(int)
    X = df.drop(columns=["admit_label"])

    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("ohe", make_ohe())]), cat_cols),
        ]
    )

    candidates = {
        "LogReg_ElasticNet": LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            max_iter=5000,
            n_jobs=-1,
            class_weight="balanced",
            random_state=seed,
        ),
        "HistGB": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_depth=6,
            max_iter=400,
            random_state=seed,
        ),
    }

    results = []
    trained = {}

    for name, clf in candidates.items():
        pipe = Pipeline([("prep", preprocess), ("model", clf)])
        pipe.fit(X_train, y_train)

        p_val = np.clip(pipe.predict_proba(X_val)[:, 1], 1e-15, 1 - 1e-15)
        best_t, best_f1 = find_best_threshold(y_val.values, p_val)

        p_test = np.clip(pipe.predict_proba(X_test)[:, 1], 1e-15, 1 - 1e-15)

        res = {
            "model": name,
            "roc_auc": float(roc_auc_score(y_test, p_test)),
            "pr_auc": float(average_precision_score(y_test, p_test)),
            "logloss": float(log_loss(y_test, p_test)),
            "brier": float(brier_score_loss(y_test, p_test)),
            "val_best_f1": float(best_f1),
            "threshold": float(best_t),
        }
        results.append(res)
        trained[name] = pipe

    best = sorted(results, key=lambda d: (d["roc_auc"], d["pr_auc"]), reverse=True)[0]
    best_pipe = trained[best["model"]]

    # Calibration (compat across sklearn versions)
    try:
        calibrator = CalibratedClassifierCV(estimator=best_pipe, method="isotonic", cv="prefit")
    except TypeError:
        calibrator = CalibratedClassifierCV(base_estimator=best_pipe, method="isotonic", cv="prefit")

    calibrator.fit(X_val, y_val)

    bundle = {
        "model": calibrator,
        "threshold": float(best["threshold"]),
        "feature_columns": X.columns.tolist(),
        "meta": {"seed": seed, "rows": int(len(df)), "results": results, "best_model": best["model"]},
    }

    joblib.dump(bundle, MODEL_PATH)
    return bundle


@st.cache_resource
def load_or_train():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return train_and_save(seed=42)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Admissions Predictor", page_icon="🎓", layout="wide")
st.title("🎓 Admissions Probability Predictor")

bundle = load_or_train()
model = bundle["model"]
THRESHOLD = float(bundle.get("threshold", 0.5))
FEATURE_COLS = bundle.get("feature_columns", None)

st.sidebar.header("Model Info")
st.sidebar.write(f"**Decision threshold:** `{THRESHOLD:.3f}`")
if "meta" in bundle:
    st.sidebar.write(f"**Best model:** `{bundle['meta'].get('best_model')}`")
    st.sidebar.write(f"**Rows trained:** `{bundle['meta'].get('rows')}`")

if st.sidebar.button("Retrain model"):
    load_or_train.clear()
    bundle = load_or_train()
    st.rerun()


def align_cols(df_in: pd.DataFrame) -> pd.DataFrame:
    df_feat = add_engineered_features(df_in)
    if FEATURE_COLS is None:
        return df_feat
    out = df_feat.copy()
    for c in FEATURE_COLS:
        if c not in out.columns:
            out[c] = np.nan
    return out[FEATURE_COLS]


def predict_proba(df_in: pd.DataFrame) -> np.ndarray:
    X = align_cols(df_in)
    p = model.predict_proba(X)[:, 1]
    return np.clip(p, 1e-15, 1 - 1e-15)


tab1, tab2, tab3 = st.tabs(["🧍 Single Prediction", "📄 Batch Scoring", "📊 Evaluate (optional)"])

with tab1:
    st.subheader("Single Prediction")

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
        ielts = st.number_input("ielts", 0.0, 9.0, 7.0, 0.5)

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
        "sop_score": sop_score, "lor_score": lor_score,
        "portfolio_score": portfolio_score, "interview_score": interview_score,
        "gender": gender, "income_bucket": income_bucket, "first_gen": first_gen
    }])

    if st.button("Predict"):
        p = float(predict_proba(row)[0])
        st.metric("Predicted probability", f"{p:.3f}")
        st.write("**Decision:**", "✅ Admit" if p >= THRESHOLD else "❌ Reject")

with tab2:
    st.subheader("Batch Scoring")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        df_up = pd.read_csv(up)
        st.dataframe(df_up.head(10))

        p = predict_proba(df_up)
        scored = df_up.copy()
        scored["pred_prob"] = np.round(p, 6)
        scored["pred_label"] = (p >= THRESHOLD).astype(int)

        st.dataframe(scored.head(10))

        buf = io.StringIO()
        scored.to_csv(buf, index=False)
        st.download_button("⬇️ Download scored CSV", buf.getvalue(), "scored_predictions.csv", "text/csv")

with tab3:
    st.subheader("Evaluate (optional)")
    st.write("Upload a CSV with `admit_label` to compute metrics.")
    up_eval = st.file_uploader("Upload labeled CSV", type=["csv"], key="eval")
    if up_eval is not None:
        df_eval = pd.read_csv(up_eval)
        if "admit_label" not in df_eval.columns:
            st.error("Missing `admit_label` column.")
        else:
            y_true = df_eval["admit_label"].astype(int).values
            p = predict_proba(df_eval)

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
