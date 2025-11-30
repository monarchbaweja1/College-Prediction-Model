from train import train_and_save, Paths, add_engineered_features
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)


# -------------------------
# Config
# -------------------------
@dataclass(frozen=True)
class Paths:
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    dataset_path: Path = Path("data/dataset_with_targets_FE.csv")
    model_path: Path = Path("artifacts/admissions_model.pkl")


# -------------------------
# Synthetic dataset generator (Mockaroo alternative)
# -------------------------
def _clip(a, lo, hi):
    return np.clip(a, lo, hi)


def generate_dataset(n: int = 21534, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    terms = ["Fall 2024", "Spring 2025", "Fall 2025", "Spring 2026"]
    programs = ["MS", "MTech", "MBA", "PhD"]
    majors = ["CS", "DS", "ECE", "Mech", "Civil", "Bio", "Econ", "Finance"]
    tiers = ["Tier1", "Tier2", "Tier3"]
    tier_probs = [0.25, 0.45, 0.30]

    countries = [
        "India", "United States", "Canada", "United Kingdom", "Germany", "France",
        "Netherlands", "Sweden", "Norway", "Australia", "New Zealand", "Singapore",
        "United Arab Emirates", "Saudi Arabia", "Qatar", "Brazil", "Mexico", "Argentina",
        "Chile", "South Africa", "Nigeria", "Kenya", "Ghana", "Egypt", "China", "Japan",
        "South Korea", "Vietnam", "Thailand", "Indonesia", "Philippines", "Malaysia",
        "Pakistan", "Bangladesh", "Sri Lanka", "Nepal", "Turkey", "Russia", "Ukraine",
        "Poland", "Spain", "Italy"
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
        "application_id": np.arange(1, n + 1),
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

    # Targets
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

    df["chance_of_admit"] = np.round(chance, 4)
    df["admit_label"] = admit

    return df


# -------------------------
# Feature Engineering (adds columns)
# -------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ielts" in out.columns:
        out["ielts"] = pd.to_numeric(out["ielts"], errors="coerce").fillna(7.0)

    out["gre_norm"]   = (out["gre_total"] - 260) / 80
    out["toefl_norm"] = (out["toefl"] - 60) / 60
    out["cgpa_norm"]  = (out["cgpa_10"] - 5.5) / 4.5
    out["aw_norm"]    = out["gre_aw"] / 6.0

    out["academic_index"] = 0.55 * out["cgpa_norm"] + 0.45 * out["gre_norm"]
    out["english_index"]  = 0.70 * out["toefl_norm"] + 0.30 * (out["ielts"] / 9.0 if "ielts" in out.columns else 7/9)

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

    out["cgpa_sq"] = out["cgpa_norm"] ** 2
    out["gre_x_cgpa"] = out["gre_norm"] * out["cgpa_norm"]
    out["sop_x_lor"] = ((out["sop_score"] - 1) / 4) * ((out["lor_score"] - 1) / 4)
    out["test_strength"] = 0.6 * out["gre_norm"] + 0.4 * out["toefl_norm"]

    return out


def make_ohe():
    # sklearn compatibility: sparse_output (new) vs sparse (old)
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def find_best_threshold(y_true: np.ndarray, p: np.ndarray) -> tuple[float, float]:
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


def train_and_save(paths: Paths, seed: int = 42) -> dict:
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset (FE)
    df = generate_dataset(n=21534, seed=seed)
    df = add_engineered_features(df)

    df.to_csv(paths.dataset_path, index=False)

    TARGET = "admit_label"
    drop_cols = [c for c in ["chance_of_admit"] if c in df.columns]

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET] + drop_cols)

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

        p_val = pipe.predict_proba(X_val)[:, 1]
        p_val = np.clip(p_val, 1e-15, 1 - 1e-15)
        best_t, best_f1 = find_best_threshold(y_val.values, p_val)

        p_test = pipe.predict_proba(X_test)[:, 1]
        p_test = np.clip(p_test, 1e-15, 1 - 1e-15)

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
    best_name = best["model"]
    best_pipe = trained[best_name]

    # Calibrate (compat across sklearn versions)
    try:
        calibrator = CalibratedClassifierCV(estimator=best_pipe, method="isotonic", cv="prefit")
    except TypeError:
        calibrator = CalibratedClassifierCV(base_estimator=best_pipe, method="isotonic", cv="prefit")

    calibrator.fit(X_val, y_val)

    bundle = {
        "model": calibrator,
        "threshold": float(best["threshold"]),
        "feature_columns": X.columns.tolist(),
        "meta": {
            "seed": seed,
            "rows": int(len(df)),
            "best_model": best_name,
            "results": results,
        }
    }

    joblib.dump(bundle, paths.model_path)
    return bundle


if __name__ == "__main__":
    paths = Paths()
    train_and_save(paths, seed=42)
    print(f"Saved model to: {paths.model_path}")
    print(f"Saved dataset to: {paths.dataset_path}")
