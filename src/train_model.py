"""
train_model.py
---------------------------------
Comprehensive ML pipeline for College Admission Prediction.

Includes:
 - Data loading & preprocessing
 - Regression model comparison (Linear, Lasso, RF)
 - Calibrated classification for decision probability
 - Threshold-based cost analysis
 - Fairness metrics by subgroup
 - Interpretability (coefficients & PDP)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, brier_score_loss, accuracy_score,
    f1_score, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.inspection import PartialDependenceDisplay


# -----------------------------------------------------------
#  DATA LOADING & CLEANING
# -----------------------------------------------------------
def load_data(path="data/admission_predict.csv"):
    """Load dataset."""
    df = pd.read_csv(path)
    print(f" Data loaded successfully! Shape: {df.shape}")
    return df


def preprocess_data(df):
    """Clean columns, rename, and handle zeros."""
    df = df.rename(columns={
        "GRE Score": "GRE",
        "TOEFL Score": "TOEFL",
        "LOR ": "LOR",
        "Chance of Admit ": "Probability"
    })

    if "Serial No." in df.columns:
        df.drop("Serial No.", axis=1, inplace=True)

    cols = ["GRE", "TOEFL", "University Rating", "SOP", "LOR", "CGPA"]
    df[cols] = df[cols].replace(0, np.nan)

    print(" Columns cleaned and zeros replaced with NaN.")
    return df


def prepare_targets(df, threshold=0.8):
    """Prepare regression & binary classification targets."""
    y_reg = df["Probability"]
    y_bin = (y_reg >= threshold).astype(int)
    X = df.drop("Probability", axis=1)
    return X, y_reg, y_bin


# -----------------------------------------------------------
#  REGRESSION MODEL SELECTION
# -----------------------------------------------------------
def stratify_bins(y, n_bins=5):
    """Bin continuous y for stratified K-fold."""
    return pd.qcut(y, q=n_bins, labels=False, duplicates="drop")


def run_regression_models(X_train, y_train):
    """Run multiple regression models with CV."""
    y_bins = stratify_bins(y_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "LinearRegression": (
            Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression())
            ]),
            {}
        ),
        "Lasso": (
            Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Lasso())
            ]),
            {"model__alpha": [0.01, 0.1, 1.0]}
        ),
        "RandomForest": (
            Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(random_state=42))
            ]),
            {"model__n_estimators": [100, 200], "model__max_depth": [None, 5, 10]}
        )
    }

    results = []
    for name, (pipe, grid) in models.items():
        gs = GridSearchCV(pipe, grid, cv=cv.split(X_train, y_bins), scoring="r2", n_jobs=-1)
        gs.fit(X_train, y_train)
        results.append([name, gs.best_params_, gs.best_score_])
        print(f"✔ {name}: R²={gs.best_score_:.4f}")

    df_res = pd.DataFrame(results, columns=["Model", "Best_Params", "CV_R2"])
    os.makedirs("results", exist_ok=True)
    df_res.to_csv("results/regression_model_comparison.csv", index=False)
    return df_res


def evaluate_regression(best_model, X_test, y_test):
    """Evaluate regression model."""
    y_pred = best_model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f" RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}")

    plt.scatter(y_test, y_pred, alpha=0.6, edgecolor="k")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlabel("Actual Probability")
    plt.ylabel("Predicted Probability")
    plt.title("Regression: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig("results/regression_actual_vs_predicted.png")
    plt.close()


# -----------------------------------------------------------
#  CALIBRATED CLASSIFICATION
# -----------------------------------------------------------
def train_calibrated_classifier(X_train, y_train):
    """Train calibrated random forest classifier."""
    clf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=42))
    ])
    grid = {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 5, 10]}

    gs = GridSearchCV(clf, grid, cv=5, scoring="roc_auc", n_jobs=-1)
    gs.fit(X_train, y_train)
    print(f" Best Classifier Params: {gs.best_params_}, CV ROC-AUC: {gs.best_score_:.4f}")

    # Version-safe calibration
    try:
        calib = CalibratedClassifierCV(estimator=gs.best_estimator_, method="isotonic", cv=5)
    except TypeError:
        calib = CalibratedClassifierCV(base_estimator=gs.best_estimator_, method="isotonic", cv=5)

    calib.fit(X_train, y_train)
    return calib


def evaluate_calibration(calib, X_test, y_test):
    """Evaluate calibrated classifier and plot calibration curve."""
    proba = calib.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    print(f"ROC-AUC={auc:.4f} | Brier Score={brier:.4f}")

    CalibrationDisplay.from_predictions(y_test, proba, n_bins=10)
    plt.title("Calibration Curve (Isotonic)")
    plt.tight_layout()
    plt.savefig("results/calibration_curve.png")
    plt.close()
    return proba


# -----------------------------------------------------------
#  FAIRNESS & THRESHOLD ANALYSIS
# -----------------------------------------------------------
def threshold_fairness(proba, y_true, X_meta, thresholds=(0.3, 0.5, 0.7)):
    """Threshold sweep & subgroup fairness report."""
    best_thr, best_util = None, -1e9
    report_lines = []

    for thr in thresholds:
        y_pred = (proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        util = -(fp*1.0 + fn*5.0)
        acc, f1 = accuracy_score(y_true, y_pred), f1_score(y_true, y_pred)
        report_lines.append(f"thr={thr:.2f} → acc={acc:.3f}, f1={f1:.3f}, util={util:.1f}")
        if util > best_util:
            best_util, best_thr = util, thr

    report_lines.append(f"\n Recommended threshold={best_thr:.2f}\n")

    # Subgroup fairness by Research
    if "Research" in X_meta.columns:
        for val in sorted(X_meta["Research"].unique()):
            mask = X_meta["Research"] == val
            acc = accuracy_score(y_true[mask], (proba[mask] >= best_thr).astype(int))
            report_lines.append(f"Research={val}: acc={acc:.3f}")

    with open("results/threshold_fairness_report.txt", "w") as f:
        f.write("\n".join(report_lines))

    print("\n".join(report_lines))
    return best_thr


# -----------------------------------------------------------
#  INTERPRETABILITY
# -----------------------------------------------------------
def interpret_linear_model(model, X):
    """Save linear regression coefficients."""
    if "model" in model.named_steps and hasattr(model.named_steps["model"], "coef_"):
        coef_df = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": model.named_steps["model"].coef_
        }).sort_values("Coefficient", ascending=False)
        coef_df.to_csv("results/linear_coefficients.csv", index=False)
        print(" Saved linear coefficients.")


def plot_pdp(model, X):
    """Partial dependence plots for GRE and CGPA."""
    for feat in ["GRE", "CGPA"]:
        if feat in X.columns:
            fig, ax = plt.subplots()
            PartialDependenceDisplay.from_estimator(model, X, [feat], ax=ax)
            plt.title(f"PDP: {feat}")
            plt.savefig(f"results/pdp_{feat}.png")
            plt.close()


# -----------------------------------------------------------
#  MAIN PIPELINE
# -----------------------------------------------------------
def main():
    df = load_data()
    df = preprocess_data(df)
    X, y_reg, y_bin = prepare_targets(df)

    # Split data
    X_train, X_test, y_reg_train, y_reg_test, y_bin_train, y_bin_test = train_test_split(
        X, y_reg, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )

    # Run regression models
    reg_results = run_regression_models(X_train, y_reg_train)
    best_model_name = reg_results.loc[reg_results["CV_R2"].idxmax(), "Model"]
    best_model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression() if best_model_name == "LinearRegression" else RandomForestRegressor())
    ])
    best_model.fit(X_train, y_reg_train)

    # Evaluate regression
    evaluate_regression(best_model, X_test, y_reg_test)

    # Interpret linear model + PDP
    interpret_linear_model(best_model, X)
    plot_pdp(best_model, X)

    # Calibrated classifier
    calib = train_calibrated_classifier(X_train, y_bin_train)
    proba = evaluate_calibration(calib, X_test, y_bin_test)

    # Threshold & fairness
    threshold_fairness(proba, y_bin_test, X_test)

    # Sample predictions
    sample = [[337, 118, 4, 4.5, 4.5, 9.65, 0]]
    pred = best_model.predict(sample)[0]*100
    with open("results/sample_predictions.txt", "w") as f:
        f.write(f"Prediction 1 (UCLA): {pred:.2f}%\n")
    print(f"\n Predicted chance of UCLA admission: {pred:.2f}%")

    print("\n Pipeline Complete — all results saved to /results/.")


if __name__ == "__main__":
    main()
